# Composed prepared-ASEC binding: review-fix lane

Session journal (history, not state — see `CLAUDE.md` "Root journals are history").
Fresh child of `origin/main` c1b83241, fast-forwarded to the reviewed head
`24f32366` (`us-composed-asec-bindings-20260906`), branch
`us-composed-asec-review-fixes-20260906`. The author's own worktree
(`_worktrees/microcosm-us-composed-asec-bindings-20260906`) and every
integration tree are untouched.

## Scope

Implement the four findings of the Fable 5.1 code review captured at
`~/architecture-reviews/microcosm-20260904/us-composed-asec-bindings-fable51-capture.md`
(context: `…/us-composed-asec-bindings-review-context.json`).

| Finding | Substance |
| --- | --- |
| code-1 | `ASEC_DEMOGRAPHIC_SOURCE_APPEARED` is keyed on a fixture-only fact and blocks the first genuine run |
| code-2 | `source_origin` is outside the CREATE producer-equality set the doc claims covers it |
| code-3 | the lane journal's State/Next are stale against the frozen commit |
| code-4 | one non-typed refusal path in `bind_composed_asec_arm_rows` |

No push, no remote merge, no deployment, no cloud, no model upgrade, and no
genuine microdata execution: source **inventory and metadata only** were read to
settle code-1.

## State

**Complete at `126a359a`.** All four findings are implemented, documented and
covered. `test_us_graph_composed_asec_binding.py`,
`test_us_spine_blindness.py` and `test_us_graph_implementation.py` run
597 passed / 0 failed / 0 errors / 0 skipped; `ruff check .` and
`tools/ci_test_groups.py --verify` are clean. The nine new regressions all fail
against the reviewed head's source, so they are regressions rather than
restatements. Nothing was pushed and no genuine run was executed. The full
report is
`~/architecture-reviews/microcosm-20260904/us-composed-asec-review-fixes-20260906/REPORT.md`.

## Established by reading (observed, not assumed)

The genuine 505-household run
`_buildo-runtime/out/us-graph-launch-20260905/asec-income-v3-505hh-ab10f778-v1`
(`run-receipt.json`: `head ab10f778…`, `status PASS`, `release_eligible false`)
carries both the ten-file prepared source directory it read and the CREATE
artifacts it produced. Metadata only was read from them — no source row, no
money/housing/income payload, and no digest is reproduced here.

- **Prepared-source parent, `asec_raw_stage.checkpoint.h5`** (the raw-stage
  parent the prepared arm is built from): its checkpoint metadata declares a
  **176-column person table that contains both `A_SEX` and `P_SEQ`**.
- **The CREATE `frame_context` artifact of that run** — the very payload the
  bind node reads as `asec_frame_context` — declares a **179-column person
  roster that likewise contains `A_SEX` and `P_SEQ`** (and `A_AGE`).
- The prepared receipt of that run declares the exact ten-file roster
  `asec_prepared_source.PREPARED_SOURCE_FILES` requires, and
  `entity_rows.person = 432523`, `entity_rows.household = 168852`.

So the reviewed refusal `ASEC_DEMOGRAPHIC_SOURCE_APPEARED:is_female` is not a
hypothetical: on genuine prepared bytes it fires at bind time and no corrected
leaf or reported observation is ever produced. The prior journal line asserting
`A_SEX` absence was established against the **invented** fixture parent, not
against source.

## Done

- [x] fresh worktree from `origin/main`, fast-forwarded to `24f32366`
- [x] read the review capture, review context, root `CLAUDE.md`, repo
      `CLAUDE.md`/`AGENTS.md`, both new modules, the binding doc, the inventory
      and guard changes, and the new test module
- [x] `uv sync --all-packages` in this child (exit 0)
- [x] genuine roster established from existing inventory/metadata (above)

- [x] code-1: truthful `present_in_prepared_arm`/`absent_from_prepared_arm`;
      refuse only an attempted write (`ASEC_DEMOGRAPHIC_COLUMN_BOUND`)
- [x] code-2: `source_origin` inside the CREATE producer equality, origin
      content recomputation retained
- [x] code-4: typed `ARM_ROWS_BUFFER_ROSTER` on wrong type, count or entries
- [x] code-3: lane journal corrected in place, `A_SEX` line included
- [x] docs: fixture metadata and observed source metadata separated
- [x] nine focused regressions, plus the guard and inventory suites

## Next

Nothing in this lane. Open decisions for the author:

- whether the binding artifact's schema version should move for the changed
  `demographics.unbound` shape (not bumped here: no stored document exists);
- reconciling this pass with the author's own branch
  `us-composed-asec-bindings-20260906`, which advanced three commits past the
  reviewed head over the same four files while this lane ran. That tree was
  read only and never modified.

## Interface changes

New public symbols `UNBOUND_DEMOGRAPHIC_COLUMNS` and
`refuse_unbound_demographic_writes`; `ASEC_DEMOGRAPHIC_SOURCE_APPEARED` removed;
`ASEC_DEMOGRAPHIC_COLUMN_BOUND`, `BINDING_UNBOUND_DEMOGRAPHICS` and
`ARM_ROWS_BUFFER_ROSTER` added; `NODE_EVIDENCE_PRODUCER` and
`ARTIFACT_PRODUCER_SPLIT` widened to `source_origin`. Only
`composed_asec_binding_v1`'s stage identity moves — the other eight US stage
hashes are byte-identical, verified by computing `implementation_hash()` in
both worktrees. The implementation inventory is unchanged, because no
non-stdlib import and no unbound symbol use was added. No genuine runner was
touched.
