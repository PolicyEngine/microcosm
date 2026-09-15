# PR triage lane journal (2026-09-12 → 2026-09-14)

Read-only triage of every open PR in PolicyEngine/microcosm. GitHub is read
only; no tracked file is edited; commits land on the local ref
`triage/journal-20260912` via plumbing so the shared worktree HEAD never moves.

## State

- Output: `~/PolicyEngine/_recovered/scratch-backup/893/triage/out-pr-triage.md`
  (the `-o` path, confirmed from `brief-pr-triage.md`) and
  `~/PolicyEngine/_recovered/scratch-backup/893/triage/pr-dispositions.json`.
- Scratch: session scratchpad holds `open_prs.json`, `merged_prs.json`,
  `closed_unmerged_prs.json`, `pr/<n>.json`, `checks/<n>.txt`, `packets/<n>.md`.
- **Run 3 (2026-09-14) is the live attempt.** Runs 1–2 (2026-09-12) died on
  model limits; their evidence is salvaged under `scratchpad/prior/` and the 16
  completed evidence-agent verdicts under `scratchpad/prior_results/`.

## Done (runs 1–2, 2026-09-12 — historical, re-verified in run 3)

- Listed 43 open PRs, 444 merged PRs (305 since 2026-07-01), 16 closed-unmerged.
- Pulled full `gh pr view --json` for all 43; ran `gh pr checks` for all 43.
- Fetched all 43 heads to `refs/heads/triage/<n>`; SHAs verified against GitHub.
- Per PR: merge-base vs its base branch, ahead/behind main, changed-file set,
  overlap with #893's then-404-file set, `git cherry` patch-equivalence (0 of 43
  PRs had any patch-equivalent commit on main), added-file presence on main
  (raw or via the populace->microcosm rename), review threads (0 for all 43),
  linked-issue states (99 numbers resolved).
- `packages/populace-build/` is absent on main; PRs #153-#629 (17 PRs) still
  target it.
- #893 split into 96 per-commit file lists.
- 16 of 43 evidence agents returned before the limit hit.

## Done (run 3, 2026-09-14)

- Salvaged runs 1-2 scratch + the 16 completed verdicts.
- `git fetch origin`: main is now `18271b28d` (merge of #904). #912 (Amendment
  20) and #904 both merged since run 1, so **every** run-1 overlap/merge-base
  number is stale and is being recomputed.
- Open-PR set is now **50, not 43**: new since 9/12 are #914 #916 #918 #921
  #923 #924 #926 #927 #928; #904 and #912 left the set by merging.
- #893 itself moved: 404 changed files on 9/11 -> 691 on 9/14, so its overlap
  set is recomputed from the new head.

### Mechanical evidence complete for all 50 (2026-09-14 ~18:50 ET)

- `gh pr view --json` refreshed for all 50; `mergeable`/`mergeStateStatus` now
  computed for every one (run 1 left 35 at UNKNOWN). Main moving turned many
  MERGEABLE rows CONFLICTING: 22 of 50 are now CONFLICTING/DIRTY.
- All 50 heads refetched to `refs/heads/triage/<n>`; every local SHA verified
  equal to `headRefOid`.
- `gh pr checks` + head-attached `statusCheckRollup` for all 50. CI eras:
  10 LEGACY (only the retired `test/wheels` quartet - stale green),
  11 NONE (no checks on the head at all), 29 CURRENT.
- Recomputed per PR against `origin/main=18271b28d`: merge-base, ahead/behind,
  net changed files, overlap with #893's **new 641-file** set, `git cherry`
  patch-equivalence (again **0 of 50** have a patch-equivalent commit on main),
  added-file presence on main raw/under the populace->microcosm rename, and the
  count of paths still under the pre-rename `packages/populace-build/` tree.
- #893 is now **rebased onto current main** (merge-base == `18271b28d`,
  0 behind, 281 commits ahead, 641 files in the net diff; GitHub reports 691
  changedFiles - the gap is GitHub's per-commit union vs the net diff).
  Its 11 named `Stage integration layer NN` commits are indexed, plus 270 later
  commits; 877 distinct files mapped to the commit that touches them.
- #906 was **retargeted to main** after #904 merged and is now MERGEABLE/CLEAN
  with `ci-ok` SUCCESS - the brief's "#906 is based on #904's branch" is stale.
- 90 of the 95 issue numbers referenced by PR bodies resolved; 5 (#1791 #9306
  #9311 #9346 #9405) are not issues in this repo (they are policyengine-us /
  policyengine-uk numbers).
- Packets rebuilt for all 50 at `scratchpad/packets/<n>.md`.

### Judgment fan-out + independent owner spot checks (2026-09-14 ~19:10 ET)

- Workflow `wf_6f60b8fc-b4d`: 50 evidence agents -> lens-specific refuters
  (close-class PRs get unique-work / superseder / dependents; MERGE-READY gets a
  readiness audit + supersession hunt; KEEP-BLOCKED-ON gets a blocker-reality
  check + supersession hunt) -> a per-PR adjudicator -> two coherence critics.
- Independent owner spot checks recorded in `scratchpad/owner_spotchecks.md`
  (verified, not delegated): #752 superseded by 05d254aa2/#756; #817, #749,
  #642, #888, #167, #413 unique on main; #153 superseded by #904's
  `dft_bus_value_anchors.json` + #730; #181 by #714's `cash_isa`; the QBI stack
  splits (#532's v1 port duplicated by #893's `puf_qbi_model.py`, but #534-#538's
  v2/v3 SSTB-crosswalk / SCF / SOI content exists nowhere); #918 adds amendments
  **25 and 26** (main is already at 24, not 20) and its `weight_update.py` is
  byte-identical to #901's while #893 lacks the file entirely.
- **#914 is the one real fold**: 12 of its 20 files are byte-identical to #893,
  including `calibrate/target_snapshots.py` and its test, and #893 commit 136 is
  literally "Merge the completed #908 snapshot slice". Its red CI is verified as
  stale country-bundle spec digests
  (`test_spec_engine_country_bundles.py:86` spec_sha256 mismatch x3).
- Byte-compared every 893-overlapping file of all 29 overlapping PRs: only #914
  has substantive identity. Filename overlap alone folded nothing.

### More owner spot checks (2026-09-14 ~19:35 ET)

- **#768 must not be closed**: the constraint it removes,
  `weeks_requires_positive_unemployment_compensation`, is still live in 8 files on
  main and in #893. Its only *added* symbol (`replay_checkpoint`) is already on
  main, so a symbol-only screen would mislabel it.
- **#873 splits**: its own "Amendment 19: typed artifacts and stable draw
  coordinates" is superseded by main's merged Amendment 19 (#911) and 20 (#912) -
  `ArtifactType` (10 files), `artifact_outputs` (11), `artifact_inputs` (12),
  `keyed_uniform` (5), `SeedSource.KEYED` (4) are all on main, and main's
  `graph/artifact_edges.py` (210 lines) and `graph/randomness.py` (78) are
  supersets of #873's (164 / 69). Unique and unmerged: the fit-layer
  `graph_apply/graph_train/_graph_qrf/graph_models` plus
  `build/transfer_example.py` (~990 lines), on neither main nor #893.
- **Frozen-interface sequencing collision**: #918 and #901 both propose lock
  `11c2abd7…/2df6cc5b…` with amendments 25 and 26; #873 proposes a different lock
  `8229270f…/07691fb5…`. Nothing else among the 50 touches decl.py or kernel.py.
  They cannot be gated in parallel.
- **#916** is green and clean but has **no `changelog.d/` fragment** among its 36
  files, and its first-time-contributor author's request for review from María is
  unanswered.
- **#718 is not superseded**: main still hardcodes `RELEASE_NAMESPACE =
  "buildo_acs_local"`; none of #923/#924/#926 carries #718's campaign derivation.
- **ACS-hours cluster mapped**: issue #765 (open) is the defect; #923 is one route
  (retain WKHP + a spine gate), #924 -> #926 is a second, explicitly chained route
  (#926's body: "the accepted full ACS recovery sidecar from #924"). #718 shares 3
  files with #923 but delivers a different feature. Pairwise file overlap among
  #924/#926/#923 is zero.
- **#325, #391, #853 carry unique work**: `ai_exposure`/`exposure_crosswalk`/
  `occupation_target` -> 0 files on main and 0 in #893; main has `ledger_artifact.py`
  and `ledger_targets.py` but no `ledger_schema.py` and no
  `schemas/consumer_fact.v1.schema.json`; `root_identity`/`raw_input_hash` -> 0 on main.

## Next (run 3)

- Finish the fan-out (28/50 evidence in at 19:40; verify + adjudicate to follow).
- Reconcile every agent verdict against `scratchpad/owner_spotchecks.md`.
- Write `out-pr-triage.md` and `pr-dispositions.json`.

## Run 4 (2026-09-14, evening) — the completing run

### Baseline re-verified, not re-assumed

- `git fetch origin` (exit 0): `origin/main` is **still `18271b28d`** (merge of
  #904, 2026-09-14 15:53 +0200). Main did **not** move between run 3 and run 4,
  so every run-3 mechanical number measured against `18271b28d` is still live.
- Open-PR set re-listed: **50**, identical numbers to run 3.
- Head drift check: 49 of 50 heads unchanged; **only #928 moved**
  (`9edfb2707` → `a4e7e131c`). Refetched. All 50 `refs/heads/triage/<n>` now
  equal `headRefOid` exactly (verified programmatically, zero mismatches).

### Refreshed for run 4 (`scratchpad/r4/`)

- `gh pr view --json` (25 fields incl. body, commits, comments, reviews,
  closingIssuesReferences, statusCheckRollup, files) for all 50 — 50/50 exit 0.
- `gh pr checks` for all 50 (28 exit 0, 19 exit 1, 3 exit 8 — non-zero is
  gh's "not all green / no checks", captured as evidence not error).
- `mech.py` re-run against `origin/main=18271b28d`: merge-base, ahead/behind,
  net file set, `--name-status`, overlap with #893's **641-file** set,
  `git cherry`, per-added-file presence on main (raw or populace→microcosm
  rename), populace-path count. **`cherry_equiv=0` for all 50** — not one open
  PR has a patch-equivalent commit on main.
- `meta.json`: author, draft, base, dates, ages, size, mergeable/mergeState,
  reviewDecision, comment counts, linked issues.
- `checks_summary.json`: CI era per PR — 11 NONE (no checks on the head at all),
  10 LEGACY (only the retired 4-check `test/wheels` quartet, stale green),
  29 CURRENT.
- Merged corpus refreshed: 449 merged PRs, 310 since 2026-07-01; latest merged
  numbers 904/909/910/911/912/913/917/919.

### Base-branch facts established this run

- Six PRs do not target `main`: the QBI chain **#534→#532, #535→#534,
  #536→#535, #537→#535, #538→#536**, and **#901→#893**.
- **#339's base `medicaid-chip-take-up` is the head branch of PR #334, which
  MERGED 2026-07-09.** The branch was never deleted, so #339 reports
  MERGEABLE/CLEAN against a base that left the repo's history two months ago.
  That green is meaningless.
- All six non-main base branches still exist on origin.

### Salvaged from run 3 (not re-derived)

- 30 of 50 evidence-agent verdicts (`wf_collected.json`); 20 missing:
  734 749 822 824 853 873 888 893 896 901 906 914 916 918 921 923 924 926 927 928.
- `owner_spotchecks.md` (18 KB of owner-verified findings) and the byte-level
  fold check `fold_check.json` / `symsweep.json`.

### Next

- Build run-4 packets, run the judgment fan-out for all 50 (fresh evidence for
  the missing 20, re-verification for the 30), adversarial verify, adjudicate.
- Write `out-pr-triage.md` + `pr-dispositions.json`.

### Run-4 owner checks, run independently of the agents (~20:30 ET)

Recorded in full at `scratchpad/r4/owner_notes_run4.md`. Headlines:

- **#739's red is a real but small packaging break.** `gh run view 33809721645
  --log-failed`: 5 FAILED in `test_us_exact_k_ladder_launcher.py`, all
  `ModuleNotFoundError: No module named 'tables'`, **wheels lane only**. Cause read
  from the diff: #739's new `_write_packaged_release` helper calls
  `pd.DataFrame(...).to_hdf(..., format="table")`, needing PyTables, which the
  wheels lane's clean venv does not have. 4 of the 5 failing tests are NEW in #739.
  Fix is a `requires_us` marker or `importorskip("tables")`.
- **#896's red is its own test and is from today.**
  `test_uk_frs_spine.py::test_driver_marks_full_fixture_smoke_outputs_non_release`
  `assert 1 == 0` at line 1825, wheels 3.13+3.14, runs at 2026-09-14 21:38/21:43 —
  the freshest CI in the set.
- **Frozen-interface collision confirmed from the lock blobs.** Only #873, #901,
  #918 touch decl.py/kernel.py/the lock (#893 does NOT — its lock blob equals
  main's). #918 and #901 propose the **identical** lock `11c2abd7…/2df6cc5b…` and
  their decl.py/kernel.py/weight_update.py blobs are byte-identical to each other;
  both add amendments **25** and **26**. #873 proposes a different lock and numbers
  its amendment **19**, a slot main filled with "Typed opaque artifacts" (its list
  runs **through 24**). So #873's charter half is numbered into occupied slots.
- **ACS-hours cluster measured**: `#923∩#924 = #923∩#926 = #924∩#926 = 0`;
  `#718∩#923 = 3` files. Issue **#765 is OPEN**. Two independent routes, plus #718
  as a separate feature.
- **#718 is not superseded and its blocker cleared**: main still has
  `RELEASE_NAMESPACE = "buildo_acs_local"` hardcoded at
  `tools/build_us_acs_local_release.py:71`; #718's `release_campaign()` derivation is
  on neither main nor #923/#924/#926. Its body holds it behind **#600, which MERGED**.
- **#339's MERGEABLE/CLEAN is an artifact**: its base `medicaid-chip-take-up` is
  PR #334's head, MERGED 2026-07-09 **without** #339's commit.

### Fan-out launched (run 4)

Workflow `wf_75516d4b-2df`: 50 evidence agents → lens-specific adversarial
refuters (close-class: unique-work / superseder-reality / dependents;
MERGE-READY: readiness-audit / supersession-hunt; KEEP-BLOCKED-ON:
blocker-reality / supersession-hunt; else supersession-hunt / value-and-cost)
→ a per-PR adjudicator that re-runs the deciding command itself.

### Run-4 owner checks, second batch (~21:10 ET)

- **#156 is a clean SUPERSEDED-CLOSE with two MERGED carriers.** Its manifest half
  landed as **#690** (`51f115102`, 2026-08-14/17) — main's `uk/source_stages.json` is
  4067 lines vs #156's 1007 — and its row-wise local-geography half as **#541**
  (`eccedfb40`, 2026-07-25), which is main's `uk_runtime/local_rowwise.py`
  ("one row per (area, household)") plus `local_doctrine.py`.
- **#821/#822/#824/#826 all carry unique unmerged work**: 0 `build/nz` paths on main
  or in #893; `concept_coverage.py` and `inventory_axiom_concepts.py` absent from
  both; #824's three Belgium target/population-input files absent though the `be/`
  scaffold exists; #826's `axiom.py` is 1861 lines vs 743 on main AND in #893, with
  19 of 20 added symbols having zero files on main.
- **Symbol-presence screen finished for all 50** (`symsweep_r4.txt`). #629, #752 and
  #817 add zero new symbols; #914 has 0-nowhere / 53-only-in-893; #873 is 40-on-main
  / 18-nowhere. **#768 is the counterexample that must stay in the report**: both its
  symbols are on main and it is still not closeable, because its contribution is a
  deletion.

## Run 5 (2026-09-14, 23:46 UTC) — the completing run

Run 4's judgment fan-out died on Fable limits (journal `wf_75516d4b-2df`:
162 agents started, **90 failed**, only 18 returned). Run 5 runs on Opus 5,
salvages runs 3-4 evidence, and finishes.

### Baseline re-verified from scratch (not assumed)

- `git fetch origin` (exit 0): `origin/main` is **still `18271b28d`**
  (merge of #904, 2026-09-14 15:53 +0200). Unchanged since run 3.
- `gh pr list --state open --limit 200` (exit 0): **50** open PRs, the exact
  same numbers as runs 3 and 4.
- Head-drift check: **zero drift**. All 50 `refs/heads/triage/<n>` still equal
  GitHub's `headRefOid`, including #928's `a4e7e131c`. So runs 3-4's git-level
  numbers were measured against the same 50 heads and the same main.
- `gh pr view` (25 fields) refreshed for all 50: **50/50 exit 0**.
- `gh pr checks` for all 50: 28 exit 0, 18 exit 1, 4 exit 8.
- `mech.py` re-run at run 5 against `origin/main=18271b28d`:
  **`cherry_equiv=0` for all 50** — confirmed a third time. Not one open PR has
  a patch-equivalent commit on main.

### What moved since run 4 (19:09 → 23:46 UTC)

- **#928 CLEAN → UNSTABLE** (its CI started after run 4 sampled it).
- Nothing else changed state. No head moved, no PR opened or closed.

### Refreshed state, run 5

- Mergeability: **26 CONFLICTING/DIRTY · 17 MERGEABLE/CLEAN · 7 MERGEABLE/UNSTABLE**.
- CI era on the current heads: **23 CURRENT · 16 LEGACY · 11 NONE**.
- `ci-ok` on the 23 CURRENT-era heads: 12 SUCCESS, 4 FAILURE (#739 #896 #901 #914),
  7 with no `ci-ok` conclusion yet (#893 #923 #927 #928 still running).

### Salvaged, not re-derived

- Run 3's 30 evidence verdicts (`wf_collected.json`) + run 4's 18
  (recovered from the workflow journal) = 48 prior opinions, treated as
  prior opinion and re-verified, never as fact.
- `owner_spotchecks.md` (18 KB) and `owner_notes_run4.md` (9 KB) — findings the
  lane owner verified personally with commands.
- `fold_check.json` (byte identity of every 893-overlapping file),
  `symsweep_r4.json` (per-PR added-symbol presence on main and in #893).

### Next

- Rebuild packets from run-5 data, run the full judgment fan-out on Opus,
  adversarially verify, adjudicate, write `out-pr-triage.md` + `pr-dispositions.json`.

### Run-5 owner checks, run independently of the agents (~00:20 UTC)

All re-verified by me this run, not inherited:

- **Frozen-interface collision, read from the lock blobs.** main
  `ed0a859a…`/`dbf57c13…`; **#901 and #918 propose the identical lock
  `11c2abd7…`/`2df6cc5b…`** and both add amendments **25** and **26**;
  **#873 proposes a different lock `8229270f…`/`07691fb5…`** and numbers its
  amendment **19** — a slot main filled two days ago (main's numbered list ends
  at **24. Population observers cannot change computation**). **#893's lock blob
  equals main's**: it does not touch the frozen interface at all.
- **#901's red is its OWN new invariant, not a flake and not #893's fault
  alone.** `gh run view 34851805402 --log-failed` (14.6 MB): the string
  `KernelContext.frame_column_order must order exactly the projected columns of
  each entity` appears **816 times**, `NodeRejectedError` 612 times, across
  **46 distinct failing tests**, nearly all `test_us_graph_*`.
  `git grep -l frame_column_order` → **13 files on #901, 9 on #918, ZERO on main
  and ZERO in #893**. So amendment 26's new column-order contract rejects #893's
  existing 19-node US graph. #918 carries the same contract and is green,
  because it lands the contract without #893's US nodes. This is real
  integration work, and it sequences #918 → (#893 nodes declare order) → #901.
- **#916 fails MERGE-READY on two counts I checked myself**: `git diff
  --name-only` shows **zero `changelog.d/` fragments** among its 36 files
  despite **11 changed source `.py` files**, and its author (`hua7450`,
  first-timer) asked `@juaristi22` on 2026-09-14 17:14 for a critical review —
  `reviewDecision` is empty and there are **0 reviews**.
- **#822 is unique, and the near-miss is worth recording**: `git grep -l
  concept_coverage origin/main` returns 3 files, but all three are
  `_validate_imputation_concept_coverage` inside `spec_engine/resolver.py` — a
  different function. `build/concept_coverage.py` itself: absent from main.
- **#391 is unique, same pattern**: `consumer_fact.v1` appears in 15 main files
  as a schema *name*, but `schemas/consumer_fact.v1.schema.json` and
  `ledger_schema.py` are not in `git ls-tree origin/main`.
- **#752 superseded, verified end to end**: main's `uv.lock` already carries
  `policyengine-us 1.819.0`; `git log -S'1.819.0' origin/main -- uv.lock` →
  `05d254aa2`, whose merge is `7b90bb188 Merge pull request #756`. #752's whole
  diff is `1.764.6→1.819.0` + `core 3.26.11→3.31.0` + a test comment.
- **#153 / #181 / #156 carriers confirmed by merge commit**: `e5035414b` →
  `0f49bf1e8 Merge pull request #714`; `51f115102` → `d3c7d5e07 Merge pull
  request #690`; `eccedfb40` is the squash of **#541**. main's
  `uk/source_stages.json` is 4067 lines vs #156's 1007.
- **#914's fold re-measured**: 12 of 20 files byte-identical to #893;
  `target_snapshot` → **0 files on main, 9 in #893**; #893 commit **136** is
  "Merge the completed #908 snapshot slice and re-pin identities on the merge
  ref", with 11 more snapshot commits around it.
- **#768 re-confirmed as the do-not-close counterexample**:
  `weeks_requires_positive_unemployment_compensation` still in **8 files on main
  and 8 in #893**; #768's diff MODIFIES exactly those files to remove it.
- **#718 re-confirmed unique**: `RELEASE_NAMESPACE = "buildo_acs_local"` still
  hardcoded at `tools/build_us_acs_local_release.py:71` on main;
  `release_campaign` → **0 files on main**. Its declared blocker **#600 is MERGED**.
- **#339 re-confirmed**: `git merge-base --is-ancestor triage/339 origin/main`
  exits 1. Its base `medicaid-chip-take-up` is PR **#334, MERGED 2026-07-09**.
- **#734 / #745 / #769 / #853 all carry unique work**: every added file MISSING
  on main (17/17, 5/5, 10/11, 4/4); `qbi_passive_passthrough`,
  `work_experience_inputs`, `realized_regime`, `microdata_pins_pending` → 0 files
  on main AND 0 in #893. #734's and #745's only main hits are the engine ABI
  declaration `us/engine_abi.lock.json` — a declaration, not an implementation.
- **#642 / #888 / #675 / #167 / #413 / #325 re-confirmed unique**: `sld_local`,
  `orrery`, `_validate_late_transfer_donor_comparator_alignment`,
  `healthcare_state_target`, `snap_release_acceptance`, `ai_exposure` → 0 on main.
- **Changelog-fragment audit across all 50**: 11 PRs change source `.py` with no
  `changelog.d/` fragment — #153 #156 #241 #391 #413 #629 #642 #675 #718 #768 **#916**.
- **Linked-issue sweep**: #718's #600 MERGED (blocker cleared); #509's #507/#508
  CLOSED; #181's #180 CLOSED; #926 explicitly refs open PR #924; #918↔#901 refer
  to each other; #923 refs OPEN issue #765.
- **Collision matrix computed** (`collisions.txt`, journal files excluded):
  #901∩#918=23, #901∩#927=15, #873∩#901=12, #537∩#538=12, #906∩#921=11,
  #779∩#916=11, #896∩#901=10, #873∩#918=10.

### Fan-out status

Workflow `wf_957cf79e-597` launched: 50 evidence agents (effort high) →
lens-specific adversarial refuters → per-PR adjudicators that re-run the
deciding command. Running on Opus.

### Run-5 measured merge-order constraints (~01:10 UTC)

- `git merge-tree --write-tree --name-only` over **93 overlapping pairs**:
  **65 conflict**. Most of that is staleness, not collision, so the honest
  subset is the 9 PRs whose merge-base IS `18271b28d`:
  **#906+#921 = 8 content conflicts** (both green, both MERGEABLE/CLEAN, both
  María's — the sharpest real constraint in the ready lane), #893+#927 = 5,
  #896+#927 = 2, #893+{#896,#906,#923} = 1 each; and **CLEAN** for
  #896+#906, #896+#921, #906+#927, #921+#927, #927+#928.
- Also CLEAN and worth naming: **#893+#914** (consistent with the fold),
  **#893+#918**, **#901+#918** (the extraction is compatible with its consumer),
  **#817+#893**, **#718+{#779,#893,#901,#923}**.
- **#873+#918 = 20 conflicts, #873+#901 = 22** — the lock collision is real.
- **Axiom-adapter three-way**: `frame/adapters/axiom.py` is 743 lines on main AND
  in #893; #821 → 904, #822 → 931, #826 → 1861. They conflict pairwise.
- **#739's red verified from the log myself**: 10 ×
  `ModuleNotFoundError: No module named 'tables'` across 5 tests in
  `test_us_exact_k_ladder_launcher.py`, **wheels lane only**; its new code calls
  `.to_hdf(..., format="table")` and `tables>=3` is declared only under the
  `us`/`uk` extras (`packages/microcosm-build/pyproject.toml:37,42`).
- **#928's UNSTABLE is a fresh run, not a failure**: all 20 jobs of run
  34907027912 are `pending`, `changes` passed, nothing failed.
- **#642's green is LEGACY**: run 31749046363, only `test`+`wheels`, under the
  workflow retired 2026-08-25; 1291 commits behind.
- **#924 and #926 are both `ci-ok` SUCCESS, CURRENT era, MERGEABLE/CLEAN, 0
  behind** — drafts, and #926's body chains it to #924.

Owner priors for all 50 written to `scratchpad/r5/owner_priors.md` **before** the
agent verdicts landed, so the fan-out is cross-checked against an independent
call rather than relayed.

### Fan-out progress

`wf_957cf79e-597`: 30/50 evidence agents returned, 0 failures. Refuters queued
behind the remaining evidence agents (the pipeline fills all 16 slots with
stage-1 work first).

### The containment result (~02:00 UTC) — the cleanest finding of the whole triage

No prior run checked whether one open PR's HEAD is an ancestor of another's.
Doing it for all 50 × 12 candidate pairs settles several rows by proof rather
than by inference:

```
git rev-list --count triage/893..triage/914   -> 0     #914 ⊆ #893
git rev-list --count triage/901..triage/918   -> 0     #918 ⊆ #901
git rev-list --count triage/538..triage/532   -> 0     #532 ⊆ #538
git rev-list --count triage/538..triage/534   -> 0     #534 ⊆ #538
git rev-list --count triage/538..triage/535   -> 0     #535 ⊆ #538
git rev-list --count triage/538..triage/536   -> 0     #536 ⊆ #538
git rev-list --count triage/538..triage/537   -> 0     #537 ⊆ #538
```

- **#914 ⊆ #893, exactly.** #914's head `16fc6cb50` **is #893's commit 165**,
  "Merge shared graph consolidation into target snapshots". Zero #914 commits
  lie outside #893; #893 has 274 beyond it. The earlier "12 of 20 files
  byte-identical, 8 differ" reading was right but understated: the 8 differ only
  because #893 kept developing past #914. FOLD-INTO-893-LAYER is now provable
  with one command.
- **#918 ⊆ #901, exactly.** Zero #918 commits outside #901; #901 has 263 beyond.
  19 of their 21 shared files are byte-identical, `decl.py`, `kernel.py`,
  `graph-interface.lock` and both changelog fragments included. But here the
  containment argues the OPPOSITE of a close: #918 is green and MERGEABLE/CLEAN
  against main, #901 is red with 46 failing tests, so #918 is the merge and #901
  is what shrinks.
- **The whole QBI stack collapses into #538.** #532, #534, #535, #536 and #537
  each have **zero** commits outside #538; #538 has 37 beyond #532. So one PR
  carries all five. #537 ⊆ #538 despite being on a sibling branch, via
  `9e23c69bb Merge branch 'qbi-v3-evidence' into qbi-v3-wiring`.

Also measured: cross-PR byte-identity over every changed file of every pair
(journals, `uv.lock`, `experiments/` and `docs/evidence/` excluded). Only four
pairs share any identical blob at all — #901/#918 (19 of 21), #537/#538 (9 of
10), #893/#914 (7 of 14), #893/#901 (1 of 27, a test fixture). Nothing else in
the 50 duplicates anything else.

Independently re-verified the evidence agent's #156 finding, which beat my own
prior: **PR #634** (`486f96d48`, merged 2026-08-10, "UK local solve through the
public calibrate front door") *deleted* `uk_runtime/local_runner.py` and
`local_solver.py` in commits `ce6bf446a` and `f55f68c04`. Both are absent from
main (`git ls-tree` finds neither), and #156 modifies exactly those files. main
did not merely re-implement #156 — it deleted #156's host and replaced it with
`local_rowwise.py` + `local_doctrine.py`.

### The maintainer already signed off on two of the closes (~02:30 UTC)

Reading the PR conversations rather than only the diffs turned up the strongest
warrant in the set:

- **juaristi22 wrote on BOTH #153 and #181, 2026-08-14 09:40**: the populace tree
  is dead so a rebase is not viable, the evidence "carries over" into the named
  successor, and "**Fine to close this in favor of #685/#623 [resp. #681] unless
  you'd rather keep it open while those land.**"
- Those successors have landed. Issue **#681** is CLOSED/COMPLETED at
  **2026-08-19T14:14:39Z**, the same second PR **#714** (`0f49bf1e8`) merged.
  Issue **#685** is CLOSED (PR #730, `433c437b1`). Issue **#623** is CLOSED
  (PRs #874 `5ab1b056f` and #904 `18271b28d`, the current main tip).
- So the condition the maintainer attached is satisfied, and closing #153/#181
  carries out a decision the team already reached.

Also: **only two of the fifty carry any `reviewDecision`** — #153
CHANGES_REQUESTED (two Max reviews, 2026-07-07 and 07-09, both verifying the DfT
England anchors against the primary publication) and #739 CHANGES_REQUESTED.

I accepted the evidence agent's **#739 = KEEP-BLOCKED-ON** over my own prior of
REBASE-THEN-GATE, after verifying its two claims myself: `reviewDecision` really
is CHANGES_REQUESTED, and the PR's two Supabase migrations really are absent from
`origin/main:supabase/migrations` (which stops at `20260829000000_...`), with no
project ref recorded on main outside test files. Rebasing to green would still
not make it safe to merge.

**Agreement check**: my priors, written before any verdict landed, matched the
evidence agents on **29 of the 30 rows** where both existed. #739 was the only
difference, and the agent was right.

All late-breaking owner findings were appended to the agents' shared `CONTEXT.md`
before any refuter or adjudicator started, so the containment result, the CI root
causes and the maintainer sign-off reach every downstream stage.

### reviewDecision is a useless readiness signal here (~03:00 UTC)

Reading every PR's conversation, not just its diff, produced the finding most
likely to prevent a bad merge:

- **48 of the 50 have an empty `reviewDecision`.** Only #153 and #739 carry one
  (both CHANGES_REQUESTED). This team posts blocking findings as issue comments
  headed `## Blocking`, not as formal GitHub reviews.
- **#906 is green, MERGEABLE/CLEAN, not a draft — and has an unanswered blocking
  review from 2026-09-14 14:52.** vahid-ahmadi: "The VOA region controls and the
  Scottish council-tax controls cannot coexist on the joint surface" — with the
  VOA rows moved to region grain the Scottish country row wins alone and **the
  nine English legs are unparented**, reproduced on a synthetic surface. It is
  NOT MERGE-READY and no mechanical signal says so.
- **#901's reviewer diagnosed it five days ago, and both causes are still
  unfixed.** Cause 2 is the `frame_column_order` contract I found independently in
  the CI log. Cause 1 I had missed: `weight_update.py` is a new `microcosm.graph`
  module but `us_runtime/graph_implementation_inventory.json`'s
  `package_rosters["microcosm.graph"]` was never updated. **Verified on the
  current head: 18 entries, no `weight_update`** (#893 has 17, also without it).
  `_validate_package_roster` raises and `survey_population_preparation.py` masks
  it as `PREPARATION_ISSUANCE_REFUSED` — every ERROR in the us-p log.
- **#413's blocking comment is the author's own**: the fit gates certify
  calibration, not the release — Build K is 52/52 within 10% calibrated but
  23/52 outside 10% simulated from the exported H5.

Added to the agents' CONTEXT.md before the refuters started.
