# US reviewed-exclusion machinery lane notes

## Scope and invariants

- Branch/worktree: `battery-excl-machinery` in the assigned worktree only.
- Port reviewed-exclusion machinery, not an exclusion decision.
- The committed US register must remain empty; only the owner may approve an
  actual exclusion
  (`packages/microcosm-build/src/microcosm/build/us/by_origin_reviewed_exclusions.json:1-5`).
- Frozen battery bands, thresholds, and comparison operations are immutable
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11587-11591,12311-12486`).
- No seed or fold edit is part of this lane.
- Builds use at most the 1% sample with seed 578, remain off-chain, and never
  touch `logbook-pending-chain.txt`.
- Every mechanism statement added below will carry a source file and line
  citation.

## Evidence read

- `experiments/battery_burndown/ADJUDICATION.md` from the read-only
  `microcosm-arm-split` worktree.
- Its `adjudication.json`, filtered to the two
  `FIX-CANDIDATE` dead-QBI rows discussed by the reviewed-exclusion finding,
  plus the reviewed-exclusion support record. The finding explicitly rejects
  both as exclusion candidates and assigns them a per-target clone-role fix
  (`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/experiments/battery_burndown/ADJUDICATION.md:59-67`;
  `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/experiments/battery_burndown/adjudication.json:461-470,3815-3860,3865-3910`).

## Mechanism record

- The production US battery accepts one fully typed reviewed-exclusion policy
  snapshot plus an optional plain-date test clock; absent a supplied snapshot,
  it resolves the committed register, and it rejects any untyped policy
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11775-11790,11832-11883`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/reviewed_exclusions.py:329-380`).
- A boolean comparison with zero weighted incidence on both origins appends a
  dead-comparison failure and records `status=dead_comparison`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:12285-12324`).
- The QBI reconciliation deliberately sets both clone-0/native SSTB flags
  false on the ASEC-compatible support role
  (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1246-1264`),
  while clone provenance defines index 0 as the ASEC role and donor-detail
  clones as the PUF role
  (`packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py:370-473`).
- The UK schema-2 template requires exactly five receipt fields, canonical
  dates, and an expiry strictly after approval
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:264-349`),
  and its strict loader rejects duplicate/non-finite JSON and any schema drift
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:545-624`).
- The UK degenerate gate applies only in-force entries, fails stale/expired/
  premature entries, and reports dormant ones
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/terminal_gates.py:294-433`).
- The rest of the adjudication-named UK template declares register resources,
  preflights/loads overrides, inserts the typed policies and one evaluation
  clock into terminal evidence, verifies the declared resource bindings, runs
  used/stale/dormant projections, and attests available evidence
  (`packages/microcosm-build/src/microcosm/build/uk/gates.json:37-44,225-260`;
  `tools/build_uk_national_dataset.py:261-290,336-367,739-758,824-840`;
  `packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py:340-393,523-557`;
  `packages/microcosm-build/src/microcosm/build/uk_runtime/battery_bindings.py:304-460,553-641,703-772`;
  `packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:998-1093,1162-1228`;
  `packages/microcosm-build/src/microcosm/build/gates.py:2512-2648`;
  `packages/microcosm-build/src/microcosm/build/gate_battery.py:601-640,947-1025`).
- US stacked gate details are snapshotted and evaluator-sealed before
  `GateReport` copies them into a manifest
  (`packages/microcosm-build/src/microcosm/build/gates.py:142-177,204-309`).
- The pool writes that same gate payload into the immutable terminal receipt
  and final manifest aliases
  (`tools/build_us_multispine_pool.py:3413-3414,3451-3514,3740-3751,4097-4117,4373-4403`).

## 1% before receipt

- Source: `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r10/logbook-receipts/populace-us-2024-stacked-f001-s578-asec1688-acs15316-20260813T041921Z-23bea977/terminal-gates.json`
- SHA-256: `96f68ddac5c833cb3c6a3eb6ea3ceb0cf04a5dcbbf92ef56581479ed368a6ce5`.
- `person/puf_tax_itemization/sstb_self_employment_income_would_be_qualified[clone_0]#dead`:
  failed; ASEC/ACS incidence 0/0; nonzero rows 0/0.
- `person/puf_tax_itemization/business_is_sstb[clone_0]#dead`: failed;
  ASEC/ACS incidence 0/0; nonzero rows 0/0.
- An empty register cannot honestly change either measured value or waive
  either failure. The measurable after result must be authenticated policy and
  exact-criterion reporting while both ineligible-for-exclusion checks remain red
  (`packages/microcosm-build/src/microcosm/build/us/by_origin_reviewed_exclusions.json:1-5`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11654-11718`).

## 1% after replay

- Replayed only the terminal battery from the saved simulated checkpoint at
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r10/populace_us_2024_stacked_pool_verify.checkpoints/stacked/2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c/simulated.checkpoint.h5`.
  Its authenticated H5 SHA-256 is
  `78721fa97cecf6f6befc83fd219abbce9453542da429e80432346d4e8272eb76`
  and size is 260,260,648 bytes. Peak replay RSS was 1,544,634,368 bytes
  (about 1.44 GiB), below the 15 GiB lane ceiling.
- The saved checkpoint is the 1%/seed-578 build surface. No new build was run,
  no predecessor digest was supplied, and no pending-chain file was touched.
- Before/after by assigned exact check:

  | Exact check | Before | After | Measured delta |
  | --- | --- | --- | --- |
  | `person/puf_tax_itemization/sstb_self_employment_income_would_be_qualified[clone_0]#dead` | `dead_comparison`; incidence 0/0; rows 0/0; failed | `dead_comparison`; incidence 0/0; rows 0/0; failed; no receipt applied | None |
  | `person/puf_tax_itemization/business_is_sstb[clone_0]#dead` | `dead_comparison`; incidence 0/0; rows 0/0; failed | `dead_comparison`; incidence 0/0; rows 0/0; failed; no receipt applied | None |

- The battery had the same 127 failures before and after. Before, its details
  had no reviewed-exclusion report. After, the in-process sealed manifest
  payload carried schema 2, committed policy digest
  `581b42d0f8ffdfe5a29671fbc9be4efc839a440da2c52951229f85d407d5758d`,
  empty resolved/applied/lifecycle buckets, and evaluation date 2026-08-20.
  The runtime emits that complete report and the manifest independently checks
  its digest, criterion classification, lifecycle, and exact applied failure
  text
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7066-7322,7698-7703,11644-11718,12202-12229`).
- Numerical improvement of the two assigned checks was not possible within
  this machinery-only lane: the adjudication assigns them a separate
  per-target clone-role comparator fix and explicitly rejects either receipt.
  The measurable improvement is from no exclusion control/report to an
  authenticated empty schema-2 policy report that proves neither failure was
  waived
  (`packages/microcosm-build/src/microcosm/build/us_runtime/reviewed_exclusions.py:290-380`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7193-7322,12202-12235`).

## Work log

- 2026-08-20: began an independent completion audit after finding the assigned
  clean worktree already contained ten unpushed lane commits. Re-attempted the
  required US-extra sync first; the global cache remained sandbox-blocked and
  the writable-cache retry remained DNS-blocked at the isolated `hatchling`
  build dependency. The existing history is being preserved while the
  loader, terminal lifecycle/manifest logic, driver wiring, packaging, and
  tests are rechecked.
- 2026-08-19: initialized the lane journals before implementation.
- 2026-08-19: the required uv sync is currently environment-blocked: the
  managed sandbox denies writes to the default cache, and the writable-cache
  retry cannot resolve PyPI DNS. No dependency or lockfile change was made.
- 2026-08-19: completed the code-cited design audit and recorded the immutable
  1% baseline receipt.
- 2026-08-19: added the empty schema-2 US resource and strict typed loader.
  The loader constrains keys to exact `leg#criterion` IDs, freezes all five
  receipt fields, authenticates overrides as a path-plus-SHA pair, and seals a
  canonical content digest
  (`packages/microcosm-build/src/microcosm/build/us/by_origin_reviewed_exclusions.json:1-5`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/reviewed_exclusions.py:33-259,262-380`).
- 2026-08-19: integrated exact criterion recording into the unchanged battery
  comparators. Eligible failures are indexed as `#dead`, `#incidence`,
  `/positive|negative#quantile`, or `#categorical` while the existing failure
  messages and frozen constants remain the emitted artifacts
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11587-11608,12285-12486`).
- 2026-08-19: added lifecycle resolution. Only exact in-force failing IDs are
  removed; passing evaluated IDs become stale failures, unevaluated IDs are
  reported dormant, and expired/premature entries add failures without
  suppressing comparator failures
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11611-11718`).
- 2026-08-19: the terminal validator reconstructs eligible failing criteria
  from sealed comparison facts, verifies the policy digest and source,
  recomputes all lifecycle/anti-rot buckets, and rejects a passing battery with
  any unapplied criterion failure or stale/out-of-force receipt
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7066-7322,7698-7742`).
- 2026-08-19: the evaluator embeds the full schema-2 report in gate details,
  which the existing evaluator seal and `GateReport` manifest copy protect
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11644-11718,11931-11956,12202-12235`;
  `packages/microcosm-build/src/microcosm/build/gates.py:142-177,204-309`).
- 2026-08-19: added paired driver flags for a schema-2 override and its raw
  SHA-256, authenticated the pair before entering stacked build work, and
  refused any override on the legacy two-spine route
  (`packages/microcosm-build/src/microcosm/build/us_runtime/reviewed_exclusions.py:346-380`;
  `tools/build_us_multispine_pool.py:532-546,4471-4493`).
- 2026-08-19: threaded the resolved typed snapshot through stacked cold/resume
  execution into the fresh terminal battery
  (`tools/build_us_multispine_pool.py:2866-2882,3293-3304,4168-4173,4357-4369`).
  The policy is intentionally absent from configured input-pin and checkpoint
  identity payloads because it changes only the terminal verdict, not the
  simulation artifact
  (`tools/build_us_multispine_pool.py:1039-1175,1178-1194`).
- 2026-08-19: the workspace suite's fail-closed inventory correctly rejected
  the new runtime module until it was reviewed. The policy loader is now
  explicitly classified as a non-population-treatment module, and the pool
  tool's source-spine-blind import graph is pinned at 65 modules
  (`packages/microcosm-build/tests/test_us_spine_blindness.py:180-270,3272-3308`).
- 2026-08-20: the complete workspace suite passes at current HEAD with
  `.venv/bin/python -m pytest -q`; `.venv/bin/ruff check .` and
  `git diff --check` also pass. The isolated wheel build could not resolve its
  `hatchling` build dependency because sandbox DNS is disabled; no package or
  lockfile was changed.
- 2026-08-20: rewrote only the unpushed lane commits so the module inventory
  classification accompanies the loader commit and the graph-count pin
  accompanies the battery-import commit. Their relevant loader/country,
  spine-blindness, battery, and driver suites pass at those commit boundaries.
- 2026-08-20: adversarial review hardened the register key grammar to the
  production `entity/family/target[clone_N][/sign]#criterion` forms. It now
  refuses malformed or noncanonical clone IDs, sign/criterion mismatches,
  non-categorical joint criteria, and repeated joint members; valid scalar,
  signed monetary, and joint categorical forms remain accepted
  (`packages/microcosm-build/src/microcosm/build/us_runtime/reviewed_exclusions.py:37-104`;
  `packages/microcosm-build/tests/test_us_reviewed_exclusions.py:128-188`).
- 2026-08-20: the authenticated override is now treated as a read-only source
  for publication and checkpoint collision checks, but remains absent from
  verified simulation pins and both configured/live checkpoint identities
  (`tools/build_us_multispine_pool.py:709-838,1039-1194`;
  `packages/microcosm-build/tests/test_us_multispine_pool_tool.py:3424-3446,3666-3713`).
- 2026-08-20: added a same-leg dual-failure regression: approving exact
  incidence leaves the failed quantile red, and approving exact quantile
  leaves failed incidence red
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11644-11718,12386-12434`;
  `packages/microcosm-build/tests/test_us_stacked_spine.py:8317-8353`).
- 2026-08-20: an independent enumeration passed all 417 unique criterion IDs
  emitted by the canonical runtime target/metric surface through the hardened
  parser, while adversarial malformed forms were rejected
  (`packages/microcosm-build/src/microcosm/build/us_runtime/reviewed_exclusions.py:37-104`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:12238-12486`).
  The complete workspace suite then passed on commit `a4d90b50` with
  `.venv/bin/python -m pytest -q` (exit 0); Ruff check, Ruff format check, and
  `git diff --check` also passed. The committed register was directly asserted
  to remain schema 2 with an empty `exclusions` object.
