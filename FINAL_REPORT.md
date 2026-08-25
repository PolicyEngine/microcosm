# Final report — defensive correctness and completeness audit for #747 and #743

## Outcome

- **PR #747:** every round-1 finding survives on final live `main`
  `f9d3b4838c137e113e87452ccdeaa734b11733d2`. The priority merged defect is a
  reproduced E6/NHS identity failure after the final age-tail redistribution.
- **PR #743:** all five round-1 P0 findings survive at the live open head
  `74b4d768f7f7c83eb0593464ceb0e2a7c81ec154`.
- The prompt's premise that #743 moved past `74b4d768` is not borne out by the
  live PR, `FETCH_HEAD`, or remote-tracking ref. Commits `96e73047`,
  `971a9699`, and `e54e546b` are ancestors of `74b4d768`, so they were already
  present in round 1.
- No build, publication, promotion, production-pointer change, or source-PR
  mutation occurred.

PR #747 merged as `2807e99c`; while this audit was being finalized, `main`
advanced one commit to `f9d3b483`. GitHub's compare and commit responses show
that the sole intervening commit changes one word in `DESIGN.md`. Every
executable, resource, receipt, register, test, and experiment byte exercised by
the #747 probes is identical at the final live tip.

## PR #747 verdicts on current main

| Round-1 finding | Verdict | Refreshed evidence and citation |
|---|---|---|
| Acceptance covers 24 of the final 25 stages | **SURVIVES ON MAIN** | Production still lists 25 stages ending in `age_tail` (`tools/build_uk_frs_spine.py:93-119,845-853`), while the receipt still says “all 24 stages” and names an untracked JSON (`experiments/686-uk-spine-swap-receipts.md:330-342,552-576`). |
| E6/NHS identity fails after age-tail | **SURVIVES ON MAIN — PRIORITY** | Production allocator → persist six NHS columns → production age-tail → production E6 moved 189/400 top-coded people to 85+ and 67 to 90+; permutation identity remained true, `matches_stored_columns=false`, and all six NHS columns mismatched. ETB persists stage-time-age outputs (`packages/microcosm-build/src/microcosm/build/uk_runtime/etb_services.py:117-139,426-436`), age-tail later mutates age (`packages/microcosm-build/src/microcosm/build/uk_runtime/age_tail.py:151-196`), and E6 recomputes from final age (`tools/verify_uk_identity_stability.py:378-386,431-464`). |
| Signed register accepts arbitrary counts and opposite-direction regressions | **SURVIVES ON MAIN, NARROWED** | Missing counts and counts set to 1 both matched the +32/−12 entry with `unsigned=[]`; water `0.776937 → 0.0` still matched the signed positive-direction entry. Matching has no quantitative input (`packages/microcosm-build/src/microcosm/build/uk_runtime/signed_differences.py:124-184`; `tools/verify_uk_spine_parity.py:124-188`). Commit [`76e39f9b`](https://github.com/PolicyEngine/microcosm/commit/76e39f9b77256766d7608c1f03e12db2f8cef118) resolves entity scope only, not this finding. |
| Strict mode permits a widened share band | **SURVIVES ON MAIN** | Empty register, `strict=true`, band `0.9`, and water delta `−0.776937` returned `verdict=parity`, `strict_failure=false`, `unsigned=[]`. Any band below 1 remains accepted and in-band differences never become unsigned (`tools/verify_uk_spine_parity.py:150-188,418-426,468-504,552-567`). |
| Weighted totals are not closed-world or identity-bound | **SURVIVES ON MAIN** | Strict comparison of `{kept: 1, omitted: 2}` with `{kept: 1}` accepted an anonymous reference identity and missing candidate identity, reported `only_in_reference=[omitted]`, and returned parity with no unsigned differences. Only intersecting keys are compared and identities are echoed, not verified (`tools/verify_uk_spine_parity.py:230-274,365-383`). |
| Re-pinned evidence retains a stale 1.56.14 figure | **SURVIVES ON MAIN** | The pinned reference says `main_residence_value=0.674658` (`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:313`), while the register and ledger still say `0.6761` (`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:142`; `experiments/686-uk-spine-comparison-ledger.md:184-190`). |

The first two rows split round-1 finding 1 into the two dispositions explicitly
requested. The last row preserves the numbered fifth finding from
`REVIEW-747.md`, although it was omitted from the prompt's abbreviated list.
No #747 finding is fully resolved.

## PR #743 verdicts at its live head

| Round-1 P0 finding | Verdict | Refreshed evidence and citation |
|---|---|---|
| Documented held-run path aborts on the production target surface | **SURVIVES** | The runbook still invokes the builder (`docs/uk-national-calibration-runbook-623.md:20-36`), which still supplies neither resolver nor exclusions (`tools/build_uk_national_dataset.py:922-934`). The exact packaged no-resolver test passed while asserting the expected `RuntimeError` (`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:86-99,203-218`; `packages/microcosm-build/tests/test_uk_national_calibration.py:478-494`). |
| Rule-1 scorer cannot score or authenticate production exports | **SURVIVES** | All 397/397 packaged measures remain slash-named; a production-shaped `dwp/uc/households` target raised `No targets compiled`. Arbitrary H5s were still accepted under trusted default labels without either SHA in the receipt (`tools/score_uk_national_candidate.py:36-59,85-93,119-177,186-198`; representative target at `packages/microcosm-build/src/microcosm/build/uk/target_references.json:9532-9545`; pristine export at `packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:341-357,380-396`). |
| Boolean person-count targets collapse with `any` | **SURVIVES** | A three-child household again resolved to `[1, 0]`, not `[3, 0]`, through `bool_any_collapse_person_to_household`; eight bindings remain affected. Boolean person-to-household routing still uses `max` (`packages/microcosm-build/src/microcosm/build/uk_runtime/measure_simulation.py:23-76`), and the test covers only legitimate predicate-any semantics (`packages/microcosm-build/tests/test_uk_measure_simulation.py:80-88`). |
| Calibration seam cannot enter the ratified Logbook chain safely | **SURVIVES** | `uk-national-calibration` still derives undeclared scope `uk/national`; declared scopes remain `uk/frs` and `us` (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:52-54`; `tools/logbook.py:63-67,192-200`; `logbook/README.md:3-5,22-44`). A fresh order probe found artifact writes before record/predecessor resolution (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:201-276`), while export still checks scope agreement (`tools/logbook.py:419-438`). |
| Release-candidate membership is operator-selectable through unsigned exclusions | **SURVIVES** | The parser still accepts `--release-candidate --measure-exclusions <operator.json>` (`tools/calibrate_uk_national_dataset.py:42-61,127-184`), while the battery omits `uk_target_surface` (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:87-116`). All five packaged entries still omit approver, adjudication, approval date, and expiry (`packages/microcosm-build/src/microcosm/build/uk/calibration_measure_exclusions.json:1-30`; complete receipt contract at `packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:259-344`). |

Vahid Ahmadi's [four second-pass findings](https://github.com/PolicyEngine/microcosm/pull/743#issuecomment-5408202783)
do not supersede these five:

| Second-pass finding | Relationship |
|---|---|
| Duplicate clone IDs collapse in `_aggregate_admin_totals` | Distinct admin-anchor weighting defect; the `dict(zip(...))` overwrite remains (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:343-389`). |
| `_band_lower_edge` chooses the first sorted Ledger filter | Adjacent to target materialization but distinct from boolean-count aggregation; it remains untied to `groupby_variable` (`packages/microcosm-build/src/microcosm/build/target_materialization.py:291-318`). |
| `UKMeasureResolver.knows` makes the entity fence vacuous | Adjacent to the missing-resolver runbook finding but distinct; the known-entity disjunction remains (`packages/microcosm-build/src/microcosm/build/uk_runtime/measure_simulation.py:121-126`). |
| Injection might mutate source frames | Confirmed safe/non-finding: the adapter copies every table and link before injection (`packages/microcosm-build/src/microcosm/build/uk_runtime/ledger_targets.py:134-140`). |

## Owner-ready #747 issue drafts

### 1. Priority: merged E6/NHS identity failure

**Title:** Merged UK spine: final age-tail breaks all six stored NHS identities

**Body:**

The UK spine merged in #747 runs `etb_services` before the final `age_tail`
stage. ETB derives and persists six NHS visit/spending columns from stage-time
age (`packages/microcosm-build/src/microcosm/build/uk_runtime/etb_services.py:117-139,426-436`);
`age_tail` later rewrites top-coded `person.age`
(`packages/microcosm-build/src/microcosm/build/uk_runtime/age_tail.py:151-196`);
E6 recomputes NHS from final age and requires equality with the stored values
(`tools/verify_uk_identity_stability.py:378-386,431-464`).

Fresh reproduction on the merged executable tree, which is byte-identical at
current main: of 400 top-coded people, 189 moved to age 85+ and 67 to age 90+.
`identical_under_permutation` remained true, but `matches_stored_columns` was
false and all six NHS columns appeared in `stored_column_mismatches`.

Acceptance criteria:

- Make final age and every stored age-derived NHS value obey one coherent
  production contract.
- Add an age-tail × NHS × E6 integration regression that fails on the merged
  ordering.
- Re-run E6 on the repaired final-stage artifact and attach an identity-bound
  receipt before relying on or promoting the merged spine.

### 2. Final-candidate acceptance

**Title:** Regenerate UK spine acceptance for the exact final 25-stage candidate

**Body:**

Current main has 25 production stages ending in `age_tail`
(`tools/build_uk_frs_spine.py:93-119,845-853`), while the committed L0 receipt
still states “all 24 stages”
(`experiments/686-uk-spine-swap-receipts.md:330-342`). The referenced receipt
under `data/ukds/acceptance/686-spine-swap/` is not tracked, and the Markdown
evidence predates the value-changing final stage and SPI persisted-zero
changes. No post-round-1 commit refreshed it.

Acceptance criteria:

- Rebuild the exact final production composition after the E6/NHS repair.
- Commit disclosure-safe evidence binding candidate content SHA, stage roster
  and composition, twin-payload identity, E4–E8, and strict parity outcomes.
- Make the receipt fail if its stage roster or candidate identity differs from
  the production plan.

### 3. Signed-register enforcement

**Title:** Enforce signed UK parity counts, directions, and magnitudes in code

**Body:**

The signed register says water moved about `+0.10` and entity counts moved
person `+32` / benunit `−12`, but the verifier only matches surface, column,
expectation, and entity
(`packages/microcosm-build/src/microcosm/build/uk_runtime/signed_differences.py:124-184`;
`tools/verify_uk_spine_parity.py:124-188`). On current main, missing candidate
counts and counts set to 1 both matched the count entry with no unsigned
difference; water moving from `0.776937` to `0.0` also matched the
positive-direction entry. Commit `76e39f9b` fixes entity scope but does not
enforce the quantitative facts.

Acceptance criteria:

- Represent and validate exact expected counts or bounded count deltas.
- Represent and validate signed direction and bounded magnitude for share and
  total entries.
- Add adversarial tests for omitted counts, implausible counts, reversed
  direction, and out-of-bound magnitude.

### 4. Strict acceptance band

**Title:** Pin strict UK spine parity to the contract share band

**Body:**

`--strict` is documented as the swap-acceptance posture, but any
`--share-band` in `[0, 1)` is accepted
(`tools/verify_uk_spine_parity.py:468-504`). A fresh probe used an empty
register, `strict=true`, band `0.9`, and a `−0.776937` water-share delta; the
result was `verdict=parity`, `strict_failure=false`, and no unsigned
differences.

Acceptance criteria:

- Make strict acceptance refuse a non-contract band, or isolate overrides in a
  diagnostic mode that cannot emit an acceptance verdict.
- Bind the effective band into the receipt and add an adversarial widened-band
  regression.

### 5. Weighted-total closure and identity

**Title:** Make UK weighted-total parity closed-world and artifact-identity-bound

**Body:**

The weighted comparator checks only intersecting keys and reports one-sided
keys without failing (`tools/verify_uk_spine_parity.py:230-274`). The caller
echoes optional identities without verifying them
(`tools/verify_uk_spine_parity.py:365-383`). A strict probe
compared `{kept: 1, omitted: 2}` against `{kept: 1}`, accepted an anonymous
reference filename and no candidate identity, reported
`only_in_reference=[omitted]`, and returned parity with no unsigned difference.

Acceptance criteria:

- Require the complete expected total-key set on both sides.
- Verify both sidecars against full expected artifact/content identities and
  bind those identities into the receipt.
- Fail strict mode on every one-sided key, missing/wrong identity, and
  cross-artifact sidecar.

### 6. Stale incumbent evidence

**Title:** Correct stale 1.56.14 main-residence evidence in the 1.56.16 UK register

**Body:**

The packaged 1.56.16 parity reference records
`main_residence_value=0.674658`
(`packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:313`),
but the signed register and comparison ledger still quote `0.6761`
(`packages/microcosm-build/src/microcosm/build/uk/spine_swap_signed_differences.json:142`;
`experiments/686-uk-spine-comparison-ledger.md:184-190`). The discrepancy
survives on current main. It does not alter direction or band classification,
but it disproves the stated completeness of the 1.56.16 cross-check.

Acceptance criteria:

- Correct the register and ledger from the pinned artifact.
- Add a machine check that every quoted incumbent share agrees with the
  packaged parity reference at the declared precision.

## Verification and delivery record

- Required `uv sync --all-packages --extra us` was attempted first. A writable
  cache resolved the 123-package lock and package download then stopped on DNS.
- Focused #747 parity/register/payload/age/NHS tests exited 0. The production E6
  and adversarial parity probes produced the exact values above.
- Focused #743 seam/scorer/resolver/materializer tests and the exact expected-
  abort regression exited 0; fresh production-shaped/static probes produced
  every table verdict above.
- Tests used a sibling environment only after confirming its `uv.lock` SHA-256
  exactly matched this checkout, with this worktree's five source roots first
  on `PYTHONPATH`. Only focused tests/probes ran; no build or high-memory
  repository-wide job ran.
- Full reports are committed locally at `review/pr-747-audit` commit
  `6b01dda54389621efe7fd8c90e281d7945223e7c` and
  `review/pr-743-audit` commit
  `6743d119d54f65cec3160bf7247d77a0d7a5f09b`.
- Three authorized Git push attempts failed because the shell could not resolve
  `github.com`; the connector publication gate was not approved. Therefore the
  local review branches contain the required commits, but their remote refs
  were not updated. No unauthorized branch was pushed.
