# Progress: populace#578 increment 2

## State

- Branch: `multispine-pool-build-578`
- Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/populace-578-inc2`
- Scope: code and small synthetic-fixture tests only; no dataset downloads or
  full-data builds.
- Implementation status: complete and locally committed; no push performed.
- Remote `main` was verified through GitHub at merge commit
  `6c14a0a8590402d1805a24e55ca5f017f39dc281`.
- The sandbox blocked `git fetch` at DNS resolution. The local worktree is
  temporarily rooted at merged PR #581's exact head
  `956dc0a3dd5ceaffae0e2007d98dc951e773d389`, whose tree is the merge result.
  Rebase onto fetched `origin/main` remains a handoff prerequisite if network
  access is not restored in this lane.
- A final fetch retry after implementation failed with the same
  `Could not resolve host: github.com` error.

## Done

- Read populace#578, its governing “Scope hardening” section, and the UK parity
  audit comment.
- Read merged populace#581's contract, review note, changed-file inventory, and
  merge metadata.
- Verified GitHub's current `main` tip is `6c14a0a`.
- Created the requested branch and worktree without modifying the existing
  checkout.
- Audited the legacy ACS multispine tool's consumers. The local-release builder
  imports its H5 helpers and the published build recipe names its CLI, so the
  legacy path will remain only as a deprecated compatibility shim.
- Traced the production stage contracts from the ASEC pre-clone checkpoint and
  byte-pinned ACS PUMS acquisition through assembly, PUF cloning and transfer,
  derivation, seeding, SSI simulation, and the fixed agreement gate.
- Identified two correctness fixes required by the new ordering:
  `transfer_acs_inputs` must fill missing cells without overwriting measured
  values, and every post-assembly `Frame` reconstruction must preserve the
  #581 metadata receipt.
- Ran the focused #581 assembly, agreement, clone-routing, and AST-guard tests
  against the starting tree: 44 passed.
- Added the canonical order-bearing runtime seam:
  assemble → clone → impute → derive → seed → simulate → terminal agreement.
  The simulated formula-output view is separate from the returned input-only
  pool, and every operator boundary revalidates the immutable #581 receipt.
- Added provenance-owner reporting helpers for JSON-ready assembly receipts and
  per-entity source-channel/clone-index counts; population operators still
  receive no source-routing interface.
- Added small two-source tests for the ordered path, batched red agreement,
  receipt loss, clone-safe ID refusal, manifest counts, and default take-up
  inventory coverage. The new seam and the full spine-blindness guard pass.
- Added a public, bounded PUF donor loader that retains the existing
  source-year E00100 alignment and processed-H5 refusal contracts.
- Made the TANF/EITC seed stage pool-safe: clone-stable IDs are a rowwise
  fallback for draw keys, and assembled-pool cells already carrying values are
  never overwritten. Other unresolved take-up defaults are filled only with
  the live engine default and explicitly labeled as such; a missing
  transfer-owned flag still fails closed.
- Added fixed-batch SSI materialization on an ephemeral, receipt-preserving
  gate view. Formula-owned `ssi` is deliberately absent from the returned
  input pool.
- Added a disposable simulation projection that fills any still-null engine
  inputs from the live engine defaults solely while materializing SSI. The
  nullable pool remains untouched, and every temporary fill is receipted.
- Added the bounded ASEC pre-clone checkpoint loader: it requires the exact
  `pre_clone_enrichment` stage binding, revalidates the stored frame identity,
  and rejects non-US or invalid household-weight artifacts.
- Added an explicit spine-blind derivation stage that completes the Schedule D
  memo input at tax-unit grain without rewriting existing values, then applies
  the shared QBI identity reconciliation while preserving the assembly receipt.
- Added `tools/build_us_multispine_pool.py`: its CLI requires five explicit
  path/SHA pairs plus `--out`, rechecks packaged ACS byte pins without
  downloading, runs the fixed pool order, retains the resumable fixed-parameter
  QRF checkpoint, and writes an input-only nullable H5, terminal agreement
  diagnostics, and an authoritative manifest. Red agreement returns nonzero
  with `simulation_ready=false`; calibration remains downstream.
- Extended the AST guard transitively from the new CLI through its US runtime
  imports, with no new source-provenance owner exceptions, and rejected the
  retired late-assembly graph.
- Bound resumable primary-QRF checkpoints to the exact five verified input
  digests and sizes. A missing or changed binding now refuses stale
  predictions instead of allowing a manifest to claim different input bytes.
- Harmonized only the ACS adapter's generated lineage fields to ASEC-compatible
  dtypes before assembly. Raw `SERIALNO`, wages, nulls, and other measured ACS
  values remain unchanged; a real tiny ASEC/ACS adapter comparison found no
  remaining shared-column dtype mismatch.
- Added focused tool-boundary fixtures covering the exact CLI surface,
  refuse-before-load SHA checks, the complete ordered two-spine seam and red
  terminal gate, failure H5/diagnostic/manifest receipts, unchanged #581
  clone-bound and receipt-loss errors, and stale QRF checkpoint refusal.
- Added a wired two-spine fixture that invokes the real missing-cell ACS
  transfer entrypoint after assembly and cloning, proves existing values remain
  untouched while missing peer-spine cells are imputed, and then reaches the
  unchanged terminal agreement gate.
- Ruff passed across all 19 changed Python files, all changed files passed
  formatting and whitespace checks, and 250 focused small-fixture tests passed
  across the new pool path plus adjacent assembly, agreement, ACS, PUF QRF/tail,
  take-up, QBI, H5-shim, and checkpoint contracts.
- Confirmed the worktree is clean and every coherent implementation step is a
  local commit.

## Next

1. From a network-enabled main session, fetch `origin/main` and rebase this
   branch onto a tip at or past `6c14a0a`, then rerun the focused checks.
2. Push/open the PR from the main session.
3. Run the separately sized full-data build and calibration/k-ladder lane;
   neither was executed here.
