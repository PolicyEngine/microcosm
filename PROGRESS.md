# Progress

## State

Round 10 implementation, review, and local proof are complete on
`tail-stratum-support-652`. Smoke-r7 exposed a genuine second producer, but not
in the late transfer: the retirement source callback independently QRF-drew
clone 1 and its clone-2 tail descendant after the primary PUF/tail stage. The
repair now mirrors each source-owned ASEC clone-1 retirement result byte-exactly
to clone 2. A canonical 18-row owner matrix covers all three target/origin/clone
combinations and is content-bound into both the late schedule and tail manifest.
The unchanged terminal preservation guard validates that canonical receipt.
The first review found two enforcement-description gaps; both are fixed, and
the follow-up review is clean with no remaining actionable findings. The branch
is ready for the requested real 1% smoke-r8 rerun; no build was run locally.

## Done

- Confirmed a clean worktree on the requested branch and exact retrigger HEAD.
- Confirmed the Round 9 commits are intact immediately below the retrigger.
- Read `CLAUDE.md` and the GitNexus debugging workflow. GitNexus MCP tools are
  unavailable in this session, so the producer/call-path trace will use local
  source, tests, commit history, and the supplied build checkpoints.
- Honored the no-network constraint; no fetch, push, GitHub call, or build has
  been attempted.
- Reconstructed smoke-r7 from its log and target checkpoints. The failing
  batch-3 target bank contains draws only for 38,604 clone-0 recipients; all
  clone-1 and clone-2 slots are null, proving late transfer did not rewrite the
  failing positive-clone cells.
- Confirmed tail construction copies clone 1 byte-for-byte into clone 2 and
  overwrites only the five declared capital-gains tail leaves. The final guard
  compares live clone 2 with live clone 1 by assembly-unique source ID after
  the complete DAG; it does not use a stale pre-source snapshot.
- Identified the actual second writer: `support_role_series` classifies every
  positive clone as PUF support, so the retirement source callback predicts
  clone 1 and clone 2 as separate stochastic QRF rows and overwrites both.
- Derived the certified two-spine ownership matrix. Tuition is PUF-owned on
  positive clones and transfer-owned on native rows; education is a consume-
  only byte-exact no-op. For both retirement overlaps, the ASEC source owns
  clone 0 and the final clone-1 value, ASEC clone 2 inherits that final value,
  ACS clone 0 is transfer-owned, and ACS positive clones are PUF-owned.
- Completed the exhaustive set audit. Primary outputs intersect persisted
  post-clone source outputs, late transfer, and recipient-owned QRF outputs in
  exactly two targets: traditional IRA and self-employed pension desired
  contributions. Adding source callback pass-through/touch outputs yields the
  third audited target, qualified tuition. The corresponding tail-owned
  intersection is empty; no other implicit dual-write target exists.
- Committed red tests for the complete owner matrix, exact intersection audit,
  education byte-identity rule, retirement clone-2 mirror, and forged tail
  receipt rejection (`f0c5ce89`).
- Added the canonical content-addressed ownership artifact, bumped late schedule
  schema v14 to v15, enforced the exhaustive dual-touch intersection at import,
  and bound the receipt into the stacked tail manifest (`9ef82161`).
- Added runtime source finalization: education proves its qualified-tuition
  callback is byte-exact consume-only; retirement preserves its certified
  owner-last clone-1 QRF result and mirrors only the two overlapping columns to
  clone 2 by assembly-unique source ID (`4d1adef9`).
- Passed the full producer-DAG tests, the focused tail-manifest test, all three
  new runtime finalizer cases, and the complete multispine-pool test file.
- Classified the new data-only ownership module as a reviewed provenance owner
  and pinned the reachable runtime graph at 64 modules; #583 passes its exact
  495-test contract.
- Fixed the review findings: education verification now fails closed if the
  callback omits its consume-only tuition passthrough, and the matrix correctly
  distinguishes ASEC callback byte verification from ACS projection masking.
  Added per-action and end-to-end omission regressions (`dbac19ab`).
- Completed an independent follow-up review with no actionable findings.
- Passed the final focused suite: 130 passed.
- Passed all 225 workspace test files in seven deterministic 32-file chunks
  plus the separately graded #583 guard: 5,922 passed and 66 skipped. Chunk
  receipts were 712/1, 562/21, 779/4, 1,038/1, 773/2, 827/1, 736/36, and the
  exact #583 receipt was 495 passed.
- Passed repository-wide `ruff check`, changed-file `ruff format --check` on
  all eight changed Python files, and both committed/working-tree
  `git diff --check` checks. A whole-repository format scan still names 29
  unrelated pre-existing files; none was rewritten.
- Gradeable smoke-r8 prediction for the same deterministic 1% input: both
  retirement overlap receipts report 3,187 byte-exact clone-1-to-clone-2
  mirrors; tuition reports an ASEC consume-only byte no-op; the terminal tail
  preservation receipt passes with the canonical ownership SHA; execution
  advances beyond `assert_stacked_tail_cells_preserved` without any of the
  three overlap columns raising a clone-2 change error.

## Next

- Run the real 1% smoke-r8 build and compare the observed mirror counts and
  terminal preservation receipt with the prediction above.
- Restore this root progress file to its pre-round base state before the final
  report, preserving the complete Round 10 journal in committed history only.
