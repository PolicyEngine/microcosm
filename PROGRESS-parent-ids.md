# PROGRESS — parent-ids-export ([microcosm#884](https://github.com/PolicyEngine/microcosm/issues/884))

## State
Worktree `~/PolicyEngine/_worktrees/microcosm-parent-ids`, branch `parent-ids-export`
from `origin/main` @ 5ab1b056f. Implementation complete; full
`uv run pytest packages/microcosm-build` in flight.

## Design decisions (recorded, not obvious from the diff)
- **Resolution key.** `(PH_SEQ, A_LINENO) -> person_id`, the same pair
  `_own_children_in_household` counts against, so the release-gate identity holds by
  construction. `person_id` already exists when the stage runs (it aligns its output on it),
  so no deferred id-mapping pass is needed.
- **int64, not float64.** The PUF support clone remaps person ids up to `10**16`; float64
  stops representing integers exactly at `2**53`.
- **ACS spine reports 0.** `acs_pums` synthesizes `PEPAR1`/`PEPAR2` as a reference-person /
  spouse link for RELSHIPP 25/26/27 only — a different construct from the ASEC measured
  pointer — and an ACS-minted `person_id` is renumbered by the assembly offset after that
  mapping runs. `map_acs_native_inputs` therefore writes the declared unknown sentinel, with
  a matching `_ACS_NATIVE_INPUT_CONTRACTS` entry.
- **Never a transfer target.** The columns join `_POOL_NATIVE_COMPLETE_OUTPUTS["person"]`, so
  `pool_transfer_target_families()` never makes them QRF targets. They are not
  PolicyEngine-US variables, so `_target_encoding` would classify an id as continuous and
  hand an ACS child a fractional interpolation between two ASEC person ids. The transfer
  surface is unchanged at 118 targets, so no pinned count, battery metric, or
  `imputation.yaml` block moves.
- **Clone remap.** `puf_support` now shifts person-referencing columns with `person_id`,
  preserving `0`. Without it a clone's child pointed at the other arm's parent.
- **`person_id` is 0-based**, so one person per frame cannot be named by a 0-means-unknown
  pointer. Their children fall back to the count proxy; the summary reports
  `pointers_unnameable_at_person_id_zero` and the identity check excludes that one person.
  Widening the sentinel is a policyengine-us#9404 contract change, not this stage's call.

## Done
- `eligibility_inputs.py`: shared pointer normalization, `_parent_person_ids`, int64 storage,
  summary diagnostics, and the gate invariants (co-residence, no self-parenting, no nulls,
  `own_children_in_household` == pointer count).
- `source_stages.json` outputs + `nonnegative_outputs` + notes; regenerated `spec/sources.yaml`;
  re-pinned `FROZEN_LEGACY_RESOURCE_SHA256["source_stages.json"]`; re-pinned the US
  `spec_sha256` in `test_us_multispine_pool_tool.py` (`c3fa28a6...`).
- `acs_inputs.py` + `operator_boundary.py` native contract; `multispine_pool.py` exclusion;
  `puf_support.py` clone remap; `us_input_mass_totals` drops the id columns.
- Tests: stage derivation (incl. three-generation), clone arms, gate failures, ACS path.
- Changelog fragment `changelog.d/884-parent-ids-export.added.md`.
- `docs/us-multispine-operator-ordering.md` eligibility row updated.

## Next
- Finish the full `packages/microcosm-build` run; fix anything it turns up.
- `uv run pytest` (whole workspace) and `uv run ruff check .`.
- Draft PR, marked blocked on
  [policyengine-us#9404](https://github.com/PolicyEngine/policyengine-us/issues/9404):
  `parent_1_id`/`parent_2_id` do not exist in the locked policyengine-us 1.819.0.
