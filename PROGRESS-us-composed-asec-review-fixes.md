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

Worktree created and fast-forwarded; the genuine person roster that code-1 turns
on has been read from two independent existing artifacts. Implementation next.

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

## Next

- [ ] code-1: truthful `present_in_prepared_arm`; refuse only an attempted bind
- [ ] code-2: `source_origin` inside the CREATE producer equality
- [ ] code-4: typed refusal for a malformed arm-rows buffer roster
- [ ] code-3: correct the lane journal
- [ ] docs: distinguish fixture metadata from observed source metadata
- [ ] focused regressions + the guard and inventory checks
