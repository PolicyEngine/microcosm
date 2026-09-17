# Lane journal: lift the native build's row-count ceilings (`native-row-ceilings`)

Branch `native-row-ceilings`, from `origin/native-scale-transport` at `a64f7b733`.
PR base is `native-scale-transport`, draft, and it stays draft.

This file is a session-handoff journal. Per the repo's CLAUDE.md, its
"State"/"Next" sections are accurate when written and historical afterward —
check git and GitHub for current truth.

## The brief

The transport lane's census (`docs/us-native-scale-transport.md` §5) found that
after the 64 MiB transport ceilings were lifted, a full-source native US build
(1,587,376 source households; ~3,471,000 stacked and ~6,943,000 cloned persons)
still meets at least four row-count bounds:

- `acs_person_coverage_columns.MAX_SELECTED_ROWS` — 1,000,000
- `asec_current_money.MAX_PERSONS` — 1,000,000
- `acs_pums.MAX_EXACT_HOUSEHOLDS` / `MAX_EXACT_PERSON_ROWS` — 1,000,000
- `survey_observed_age.MAX_ROWS` — 2,000,000

and that `survey_origin_budget.MAX_GROUPS` (1,000,000) was not established.
Max decided (2026-09-17) they are lifted as **one change with one argument**,
not one build at a time.

## State

Census in progress. Nothing implemented yet.

## Done

- Worktree verified at `a64f7b733`; `uv sync --all-packages --locked --extra us` exit 0.
- Read the transport lane's §2c/§2e (the `MAX_ROSTER_BYTES` argument shape:
  an explicit resource ceiling at a fixed multiple above the full-source count)
  and its §4 (how it re-derived pins through `graph_implementation._dependency_contract`).

## Next

1. Census every `MAX_*` bound in `us_runtime/` reachable from the 19-node
   financial graph and the 45-node pilot graph, from code at this head.
2. Establish `survey_origin_budget.MAX_GROUPS`.
3. Write `docs/us-native-row-ceilings.md` — the one argument.
4. Implement; a test per moved bound.
5. Re-derive every moved pin through its generator.
6. Draft PR against `native-scale-transport`.
