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

- Census of every `MAX_*` bound in `us_runtime/` reachable from the two graph
  entry points: 146 bounds across five module families, each with its
  enforcement site, refusal code, what it protects, and its counts at 1/10 and
  full source. **14 bind at full source; 5 of those bind at 1/10 too.**
- `survey_origin_budget.MAX_GROUPS` **established**: `allocation_instructions`
  requires one instruction per selected household, so a full-source budget has
  1,587,376 groups and the 1,000,000 bound binds.
- Five constants lifted under the rule (commit `14defbfc0`).
- Pins re-derived through their generators (`2ebd246f1`): one moved,
  `acs_native_coverage_binding._ACCEPTED["acs_pums.py"]`.
- **An inherited break re-pinned** (`55ca820c7`): base-branch commit `b6081efcb`
  added `path.read_bytes()` to `_spill_roster` without regenerating
  `survey_population_preparation.py`'s `resource_accesses_sha256`, so
  `implementation_manifest()` raised for every stage containing it — including
  the one the nineteen-node path runs. Not this branch's file; re-pinned here so
  the base is functional and this lane's own manifests can be built.
- Tests: the rule as an executable table, plus a boundary test per moved bound.

## The loudest finding

`survey_origin_budget.MAX_PAYLOAD_BYTES` (64 MiB) admits **87,838 households,
5.53% of source** — below 1/10, and below the 96,860-household ceiling the
transport lane lifted. Measured through the module's own encoder at full-source
id widths: 764 B per group, 1.13 GiB at full source, 18.07× the cap. It is a
byte transport, so this lane lifts `MAX_GROUPS` and leaves it: at full source the
refusal moves from `GROUP_COUNT_BOUND` to `TRANSPORT_LIMIT`. Necessary, not
sufficient, and the report says so.

## Next

1. Finish the adversarial verification of the 14 binding verdicts.
2. Decide, on that evidence, whether the two pure row-count bounds the transport
   lane's census missed (`asec_demographic_source._MAX_PERSONS`,
   `current_child_property_income_source.MAX_ROWS`, both 600,000) move here.
3. `docs/us-native-row-ceilings.md`.
4. Tests as CI runs them; `ci_test_groups --verify`; `spec_engine_coverage --check`.
5. Draft PR against `native-scale-transport`.
