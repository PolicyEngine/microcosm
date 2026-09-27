# Lane journal: lift the native build's row-count ceilings (`native-row-ceilings`)

Branch `native-row-ceilings`, from `origin/native-scale-transport` at `a64f7b733`.
PR base is `native-scale-transport`, draft, and it stays draft.

This file is a session-handoff journal. Per the repo's CLAUDE.md, its
"State"/"Next" sections are accurate when written and historical afterward —
check git and GitHub for current truth.

## The rule (the one argument)

**A bound moves only if a full-source native build meets it, and a bound that
moves becomes four times the measured full-source count of exactly what it
counts, rounded up to the next whole million.**

Byte transports are *not* this rule's. The transport lane's answer to a byte
ceiling is a segmented stream under `MAX_ROSTER_BYTES`, not a larger single cap,
and applying this rule to one would be the wrong argument.

## The counts are measured, not extrapolated

The recovered 1/1000 pilot artifact
(`_recovered/pilot-runs/native19-required-20260912/run/financial-artifacts/preparation.json`,
sha256 `34b362d8…`) carries the whole ACS and ASEC **catalogues**, which do not
scale with the selection fraction. Its ACS occupied + institutional-GQ +
noninstitutional-GQ households plus its ASEC households equal its own
`selection.supplied_households` exactly (1,348,408 + 84,422 + 98,784 + 55,762 =
1,587,376), so "the whole catalogue" and "what a full-source selection supplies"
are the same set:

| | households | persons |
|---|---|---|
| ACS | 1,531,614 | 3,422,888 |
| ASEC | 55,762 | 142,125 |
| stacked | 1,587,376 | 3,565,013 |
| combined clone | 3,174,752 | 7,130,026 |

The transport lane's ~3,471,000 stacked / ~6,943,000 cloned were the 1/1000
per-household ratio extrapolated; these supersede them.
`experiments/native-row-ceilings/roster_census.py` re-derives them and asserts
the reconciliation.

## Done

- **Census**: 146 bounds across five module families, reachable from the 19-node
  financial graph and the 45-node pilot graph, each with its enforcement site,
  refusal code and exception type, what it protects, its counts at 1/10 and at
  full source, and whether it binds. 41 of those verdicts then went through an
  adversarial pass; no headline binding call was overturned.
  **14 bind at full source, 6 of them at 1/10.**
- `survey_origin_budget.MAX_GROUPS` **established**: `allocation_instructions`
  requires one instruction per selected household, so a full-source budget has
  1,587,376 groups against a 1,000,000 bound.
- **Seven ceilings lifted** under the rule, each with a boundary test.
- Pins re-derived through their generators: one moved
  (`acs_native_coverage_binding._ACCEPTED["acs_pums.py"]`). 124 inventory
  contracts checked, 10 stage manifests built.
- **An inherited break re-pinned**: base-branch `b6081efcb` added
  `path.read_bytes()` to `_spill_roster` without regenerating
  `survey_population_preparation.py`'s `resource_accesses_sha256`, so
  `implementation_manifest()` raised for every stage containing it. Proven
  pre-existing by recomputing the contract from `origin/native-scale-transport`,
  `a64f7b733` and `b6081efcb`'s own blobs, and by running the affected tests
  against a base worktree: 7 fail there with that exact error and all pass here.
- `docs/us-native-row-ceilings.md`, and the rule as an executable table.

## The three findings the report leads with

1. **`acs_person_coverage_authentication.MAX_BODY_BYTES` admits 12,911 selected
   ACS persons — 0.38% of source.** Measured over 200,000 real records of the
   pilot's captured public archive. The tightest ceiling on the path, 265× under
   at full source, and below 1/100.
2. **The preparation-receipt ceiling the transport lane lifted is still enforced
   one module downstream** at `PREPARATION_MAX_BYTES` = 64 MiB, which admits
   96,839 households — to within rounding the exact 96,860 that lane reported as
   lifted.
3. **`survey_origin_budget.MAX_PAYLOAD_BYTES` admits 87,838 households, 5.53%**,
   and cannot be raised at all in that module: `graph._bounded_json` refuses any
   limit above 64 MiB before encoding a byte.

All three are byte transports and take the transport lane's argument, not this
one. All three are pinned in tests.

## State

Complete. **[PR #949](https://github.com/PolicyEngine/microcosm/pull/949)**,
draft against `native-scale-transport`, MERGEABLE. CI does not run on it by
design (`test.yml` triggers on `pull_request: branches: [main]`); the local
gates in the report's section 8 are the only ones it has, and all are green:
399 passed over the seven touched test files, `verification=ok`, spec-engine
42156/42156 and 41/41, ruff clean, all three pin generators idempotent.

## Next
- Open for Max: whether the inherited re-pin stays here or moves to #945; and
  whether the byte transports above are one follow-up lane or several.
